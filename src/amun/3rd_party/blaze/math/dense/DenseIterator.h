//=================================================================================================
/*!
//  \file blaze/math/dense/DenseIterator.h
//  \brief Header file for the DenseIterator class template
//
//  Copyright (C) 2013 Klaus Iglberger - All Rights Reserved
//
//  This file is part of the Blaze library. You can redistribute it and/or modify it under
//
//  * The names of its contributors may not be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
//  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
//  OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
//  SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
//  TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
//  BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
//  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
//  DAMAGE.
*/
//=================================================================================================

#ifndef _BLAZE_MATH_DENSE_DENSEITERATOR_H_
#define _BLAZE_MATH_DENSE_DENSEITERATOR_H_


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <iterator>
#include <blaze/math/SIMD.h>
#include <blaze/util/AlignmentCheck.h>
#include <blaze/util/Assert.h>
#include <blaze/util/Types.h>


namespace blaze {

//=================================================================================================
//
//  CLASS DEFINITION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Implementation of a generic iterator for dense vectors and matrices.
// \ingroup math
//
// The DenseIterator represents a generic random-access iterator that can be used for dense
// vectors and specific rows/columns of dense matrices.
*/
template< typename Type  // Type of the elements
        , bool AF >      // Alignment flag
class DenseIterator
{
 public:
   //**Type definitions****************************************************************************
   typedef std::random_access_iterator_tag  IteratorCategory;  //!< The iterator category.
   typedef Type                             ValueType;         //!< Type of the underlying elements.
   typedef Type*                            PointerType;       //!< Pointer return type.
   typedef Type&                            ReferenceType;     //!< Reference return type.
   typedef ptrdiff_t                        DifferenceType;    //!< Difference between two iterators.

   // STL iterator requirements
   typedef IteratorCategory  iterator_category;  //!< The iterator category.
   typedef ValueType         value_type;         //!< Type of the underlying elements.
   typedef PointerType       pointer;            //!< Pointer return type.
   typedef ReferenceType     reference;          //!< Reference return type.
   typedef DifferenceType    difference_type;    //!< Difference between two iterators.

   //! SIMD type of the elements.
   typedef SIMDTrait_<Type>  SIMDType;
   //**********************************************************************************************

   //**Constructors********************************************************************************
   /*!\name Constructors */
   //@{
   explicit inline DenseIterator() noexcept;
   explicit inline DenseIterator( Type* ptr ) noexcept;

   template< typename Other, bool AF2 >
   inline DenseIterator( const DenseIterator<Other,AF2>& it ) noexcept;
   //@}
   //**********************************************************************************************

   //**Destructor**********************************************************************************
   // No explicitly declared destructor.
   //**********************************************************************************************

   //**Assignment operators************************************************************************
   /*!\name Assignment operators */
   //@{
   // No explicitly declared copy assignment operator.
   inline DenseIterator& operator+=( ptrdiff_t inc ) noexcept;
   inline DenseIterator& operator-=( ptrdiff_t inc ) noexcept;
   //@}
   //**********************************************************************************************

   //**Increment/decrement operators***************************************************************
   /*!\name Increment/decrement operators */
   //@{
   inline DenseIterator&      operator++()      noexcept;
   inline const DenseIterator operator++( int ) noexcept;
   inline DenseIterator&      operator--()      noexcept;
   inline const DenseIterator operator--( int ) noexcept;
   //@}
   //**********************************************************************************************

   //**Access operators****************************************************************************
   /*!\name Access operators */
   //@{
   inline ReferenceType operator[]( size_t index ) const noexcept;
   inline ReferenceType operator* () const noexcept;
   inline PointerType   operator->() const noexcept;
   //@}
   //**********************************************************************************************

   //**Utility functions***************************************************************************
   /*!\name Utility functions */
   //@{
   inline PointerType base() const noexcept;
   //@}
   //**********************************************************************************************

   //**Expression template evaluation functions****************************************************
   /*!\name Expression template evaluation functions */
   //@{
   inline const SIMDType load  () const noexcept;
   inline const SIMDType loada () const noexcept;
   inline const SIMDType loadu () const noexcept;
   inline void           store ( const SIMDType& value ) const noexcept;
   inline void           storea( const SIMDType& value ) const noexcept;
   inline void           storeu( const SIMDType& value ) const noexcept;
   inline void           stream( const SIMDType& value ) const noexcept;
   //@}
   //**********************************************************************************************

 private:
   //**Member variables****************************************************************************
   /*!\name Member variables */
   //@{
   PointerType ptr_;  //!< Pointer to the current element.
   //@}
   //**********************************************************************************************
};
//*************************************************************************************************




//=================================================================================================
//
//  CONSTRUCTORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Default constructor for the DenseIterator class.
*/
template< typename Type  // Type of the elements
        , bool AF >      // Alignment flag
inline DenseIterator<Type,AF>::DenseIterator() noexcept
   : ptr_( nullptr )  // Pointer to the current element
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Constructor for the DenseIterator class.
//
// \param ptr Pointer to the initial element.
*/
template< typename Type  // Type of the elements
        , bool AF >      // Alignment flag
inline DenseIterator<Type,AF>::DenseIterator( Type* ptr ) noexcept
   : ptr_( ptr )  // Pointer to the current element
{}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Conversion constructor from different DenseIterator instances.
//
// \param it The foreign DenseIterator instance to be copied.
*/
template< typename Type   // Type of the elements
        , bool AF >       // Alignment flag
template< typename Other  // Type of the foreign elements
        , bool AF2 >      // Alignment flag of the foreign iterator
inline DenseIterator<Type,AF>::DenseIterator( const DenseIterator<Other,AF2>& it ) noexcept
   : ptr_( it.base() )  // Pointer to the current element
{}
//*************************************************************************************************




//=================================================================================================
//
//  ASSIGNMENT OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Addition assignment operator.
//
// \param inc The increment of the iterator.
// \return Reference to the incremented iterator.
*/
template< typename Type  // Type of the elements
        , bool AF >      // Alignment flag
inline DenseIterator<Type,AF>& DenseIterator<Type,AF>::operator+=( ptrdiff_t inc ) noexcept
{
   ptr_ += inc;
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Subtraction assignment operator.
//
// \param dec The decrement of the iterator.
// \return Reference to the decremented iterator.
*/
template< typename Type  // Type of the elements
        , bool AF >      // Alignment flag
inline DenseIterator<Type,AF>& DenseIterator<Type,AF>::operator-=( ptrdiff_t dec ) noexcept
{
   ptr_ -= dec;
   return *this;
}
//*************************************************************************************************




//=================================================================================================
//
//  INCREMENT/DECREMENT OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Pre-increment operator.
//
// \return Reference to the incremented iterator.
*/
template< typename Type  // Type of the elements
        , bool AF >      // Alignment flag
inline DenseIterator<Type,AF>& DenseIterator<Type,AF>::operator++() noexcept
{
   ++ptr_;
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Post-increment operator.
//
// \return The previous position of the iterator.
*/
template< typename Type  // Type of the elements
        , bool AF >      // Alignment flag
inline const DenseIterator<Type,AF> DenseIterator<Type,AF>::operator++( int ) noexcept
{
   return DenseIterator( ptr_++ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Pre-decrement operator.
//
// \return Reference to the decremented iterator.
*/
template< typename Type  // Type of the elements
        , bool AF >      // Alignment flag
inline DenseIterator<Type,AF>& DenseIterator<Type,AF>::operator--() noexcept
{
   --ptr_;
   return *this;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Post-decrement operator.
//
// \return The previous position of the iterator.
*/
template< typename Type  // Type of the elements
        , bool AF >      // Alignment flag
inline const DenseIterator<Type,AF> DenseIterator<Type,AF>::operator--( int ) noexcept
{
   return DenseIterator( ptr_-- );
}
//*************************************************************************************************




//=================================================================================================
//
//  OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Direct access to the underlying elements.
//
// \param index Access index.
// \return Reference to the accessed value.
*/
template< typename Type  // Type of the elements
        , bool AF >      // Alignment flag
inline typename DenseIterator<Type,AF>::ReferenceType
   DenseIterator<Type,AF>::operator[]( size_t index ) const noexcept
{
   return ptr_[index];
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Direct access to the element at the current iterator position.
//
// \return Reference to the current element.
*/
template< typename Type  // Type of the elements
        , bool AF >      // Alignment flag
inline typename DenseIterator<Type,AF>::ReferenceType
   DenseIterator<Type,AF>::operator*() const noexcept
{
   return *ptr_;
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Direct access to the element at the current iterator position.
//
// \return Pointer to the element at the current iterator position.
*/
template< typename Type  // Type of the elements
        , bool AF >      // Alignment flag
inline typename DenseIterator<Type,AF>::PointerType
   DenseIterator<Type,AF>::operator->() const noexcept
{
   return ptr_;
}
//*************************************************************************************************




//=================================================================================================
//
//  UTILITY FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Low-level access to the underlying member of the iterator.
//
// \return Pointer to the current memory location.
*/
template< typename Type  // Type of the elements
        , bool AF >      // Alignment flag
inline typename DenseIterator<Type,AF>::PointerType DenseIterator<Type,AF>::base() const noexcept
{
   return ptr_;
}
//*************************************************************************************************




//=================================================================================================
//
//  EXPRESSION TEMPLATE EVALUATION FUNCTIONS
//
//=================================================================================================

//*************************************************************************************************
/*!\brief Load of the SIMD element at the current iterator position.
//
// \return The loaded SIMD element.
//
// This function performs a load of the SIMD element of the current element. This function must
// \b NOT be called explicitly! It is used internally for the performance optimized evaluation
// of expression templates. Calling this function explicitly might result in erroneous results
// and/or in compilation errors.
*/
template< typename Type  // Type of the elements
        , bool AF >      // Alignment flag
inline const typename DenseIterator<Type,AF>::SIMDType
   DenseIterator<Type,AF>::load() const noexcept
{
   if( AF )
      return loada();
   else
      return loadu();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Aligned load of the SIMD element at the current iterator position.
//
// \return The loaded SIMD element.
//
// This function performs an aligned load of the SIMD element of the current element. This
// function must \b NOT be called explicitly! It is used internally for the performance optimized
// evaluation of expression templates. Calling this function explicitly might result in erroneous
// results and/or in compilation errors.
*/
template< typename Type  // Type of the elements
        , bool AF >      // Alignment flag
inline const typename DenseIterator<Type,AF>::SIMDType
   DenseIterator<Type,AF>::loada() const noexcept
{
   BLAZE_INTERNAL_ASSERT( checkAlignment( ptr_ ), "Invalid alignment detected" );

   return blaze::loada( ptr_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Unaligned load of the SIMD element at the current iterator position.
//
// \return The loaded SIMD element.
//
// This function performs an unaligned load of the SIMD element of the current element. This
// function must \b NOT be called explicitly! It is used internally for the performance optimized
// evaluation of expression templates. Calling this function explicitly might result in erroneous
// results and/or in compilation errors.
*/
template< typename Type  // Type of the elements
        , bool AF >      // Alignment flag
inline const typename DenseIterator<Type,AF>::SIMDType
   DenseIterator<Type,AF>::loadu() const noexcept
{
   return blaze::loadu( ptr_ );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Store of the SIMD element at the current iterator position.
//
// \param value The SIMD element to be stored.
// \return void
//
// This function performs a store of the SIMD element of the current element. This function must
// \b NOT be called explicitly! It is used internally for the performance optimized evaluation
// of expression templates. Calling this function explicitly might result in erroneous results
// and/or in compilation errors.
*/
template< typename Type  // Type of the elements
        , bool AF >      // Alignment flag
inline void DenseIterator<Type,AF>::store( const SIMDType& value ) const noexcept
{
   if( AF )
      storea( value );
   else
      storeu( value );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Aligned store of the SIMD element at the current iterator position.
//
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an aligned store of the SIMD element of the current element. This
// function must \b NOT be called explicitly! It is used internally for the performance optimized
// evaluation of expression templates. Calling this function explicitly might result in erroneous
// results and/or in compilation errors.
*/
template< typename Type  // Type of the elements
        , bool AF >      // Alignment flag
inline void DenseIterator<Type,AF>::storea( const SIMDType& value ) const noexcept
{
   blaze::storea( ptr_, value );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Unaligned store of the SIMD element at the current iterator position.
//
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an unaligned store of the SIMD element of the current element. This
// function must \b NOT be called explicitly! It is used internally for the performance optimized
// evaluation of expression templates. Calling this function explicitly might result in erroneous
// results and/or in compilation errors.
*/
template< typename Type  // Type of the elements
        , bool AF >      // Alignment flag
inline void DenseIterator<Type,AF>::storeu( const SIMDType& value ) const noexcept
{
   blaze::storeu( ptr_, value );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Aligned, non-temporal store of the SIMD element at the current iterator position.
//
// \param value The SIMD element to be stored.
// \return void
//
// This function performs an aligned, non-temporal store of the SIMD element of the current
// element. This function must \b NOT be called explicitly! It is used internally for the
// performance optimized evaluation of expression templates. Calling this function explicitly
// might result in erroneous results and/or in compilation errors.
*/
template< typename Type  // Type of the elements
        , bool AF >      // Alignment flag
inline void DenseIterator<Type,AF>::stream( const SIMDType& value ) const noexcept
{
   blaze::stream( ptr_, value );
}
//*************************************************************************************************




//=================================================================================================
//
//  GLOBAL OPERATORS
//
//=================================================================================================

//*************************************************************************************************
/*!\name DenseIterator operators */
//@{
template< typename T1, bool AF1, typename T2, bool AF2 >
inline bool operator==( const DenseIterator<T1,AF1>& lhs, const DenseIterator<T2,AF2>& rhs ) noexcept;

template< typename T1, bool AF1, typename T2, bool AF2 >
inline bool operator!=( const DenseIterator<T1,AF1>& lhs, const DenseIterator<T2,AF2>& rhs ) noexcept;

template< typename T1, bool AF1, typename T2, bool AF2 >
inline bool operator<( const DenseIterator<T1,AF1>& lhs, const DenseIterator<T2,AF2>& rhs ) noexcept;

template< typename T1, bool AF1, typename T2, bool AF2 >
inline bool operator>( const DenseIterator<T1,AF1>& lhs, const DenseIterator<T2,AF2>& rhs ) noexcept;

template< typename T1, bool AF1, typename T2, bool AF2 >
inline bool operator<=( const DenseIterator<T1,AF1>& lhs, const DenseIterator<T2,AF2>& rhs ) noexcept;

template< typename T1, bool AF1, typename T2, bool AF2 >
inline bool operator>=( const DenseIterator<T1,AF1>& lhs, const DenseIterator<T2,AF2>& rhs ) noexcept;

template< typename Type, bool AF >
inline const DenseIterator<Type,AF> operator+( const DenseIterator<Type,AF>& it, ptrdiff_t inc ) noexcept;

template< typename Type, bool AF >
inline const DenseIterator<Type,AF> operator+( ptrdiff_t inc, const DenseIterator<Type,AF>& it ) noexcept;

template< typename Type, bool AF >
inline const DenseIterator<Type,AF> operator-( const DenseIterator<Type,AF>& it, ptrdiff_t inc ) noexcept;

template< typename Type, bool AF >
inline ptrdiff_t operator-( const DenseIterator<Type,AF>& lhs, const DenseIterator<Type,AF>& rhs ) noexcept;
//@}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Equality comparison between two DenseIterator objects.
//
// \param lhs The left-hand side iterator.
// \param rhs The right-hand side iterator.
// \return \a true if the iterators refer to the same element, \a false if not.
*/
template< typename T1  // Element type of the left-hand side iterator
        , bool AF1     // Alignment flag of the left-hand side iterator
        , typename T2  // Element type of the right-hand side iterator
        , bool AF2 >   // Alignment flag of the right-hand side iterator
inline bool operator==( const DenseIterator<T1,AF1>& lhs, const DenseIterator<T2,AF2>& rhs ) noexcept
{
   return lhs.base() == rhs.base();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Inequality comparison between two DenseIterator objects.
//
// \param lhs The left-hand side iterator.
// \param rhs The right-hand side iterator.
// \return \a true if the iterators don't refer to the same element, \a false if they do.
*/
template< typename T1  // Element type of the left-hand side iterator
        , bool AF1     // Alignment flag of the left-hand side iterator
        , typename T2  // Element type of the right-hand side iterator
        , bool AF2 >   // Alignment flag of the right-hand side iterator
inline bool operator!=( const DenseIterator<T1,AF1>& lhs, const DenseIterator<T2,AF2>& rhs ) noexcept
{
   return lhs.base() != rhs.base();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Less-than comparison between two DenseIterator objects.
//
// \param lhs The left-hand side iterator.
// \param rhs The right-hand side iterator.
// \return \a true if the left-hand side iterator is smaller, \a false if not.
*/
template< typename T1  // Element type of the left-hand side iterator
        , bool AF1     // Alignment flag of the left-hand side iterator
        , typename T2  // Element type of the right-hand side iterator
        , bool AF2 >   // Alignment flag of the right-hand side iterator
inline bool operator<( const DenseIterator<T1,AF1>& lhs, const DenseIterator<T2,AF2>& rhs ) noexcept
{
   return lhs.base() < rhs.base();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Greater-than comparison between two DenseIterator objects.
//
// \param lhs The left-hand side iterator.
// \param rhs The right-hand side iterator.
// \return \a true if the left-hand side iterator is greater, \a false if not.
*/
template< typename T1  // Element type of the left-hand side iterator
        , bool AF1     // Alignment flag of the left-hand side iterator
        , typename T2  // Element type of the right-hand side iterator
        , bool AF2 >   // Alignment flag of the right-hand side iterator
inline bool operator>( const DenseIterator<T1,AF1>& lhs, const DenseIterator<T2,AF2>& rhs ) noexcept
{
   return lhs.base() > rhs.base();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Less-or-equal-than comparison between two DenseIterator objects.
//
// \param lhs The left-hand side iterator.
// \param rhs The right-hand side iterator.
// \return \a true if the left-hand side iterator is less or equal, \a false if not.
*/
template< typename T1  // Element type of the left-hand side iterator
        , bool AF1     // Alignment flag of the left-hand side iterator
        , typename T2  // Element type of the right-hand side iterator
        , bool AF2 >   // Alignment flag of the right-hand side iterator
inline bool operator<=( const DenseIterator<T1,AF1>& lhs, const DenseIterator<T2,AF2>& rhs ) noexcept
{
   return lhs.base() <= rhs.base();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Greater-or-equal-than comparison between two DenseIterator objects.
//
// \param lhs The left-hand side iterator.
// \param rhs The right-hand side iterator.
// \return \a true if the left-hand side iterator is greater or equal, \a false if not.
*/
template< typename T1  // Element type of the left-hand side iterator
        , bool AF1     // Alignment flag of the left-hand side iterator
        , typename T2  // Element type of the right-hand side iterator
        , bool AF2 >   // Alignment flag of the right-hand side iterator
inline bool operator>=( const DenseIterator<T1,AF1>& lhs, const DenseIterator<T2,AF2>& rhs ) noexcept
{
   return lhs.base() >= rhs.base();
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Addition between a DenseIterator and an integral value.
//
// \param it The iterator to be incremented.
// \param inc The number of elements the iterator is incremented.
// \return The incremented iterator.
*/
template< typename Type  // Element type of the iterator
        , bool AF >      // Alignment flag of the iterator
inline const DenseIterator<Type,AF> operator+( const DenseIterator<Type,AF>& it, ptrdiff_t inc ) noexcept
{
   return DenseIterator<Type,AF>( it.base() + inc );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Addition between an integral value and a DenseIterator.
//
// \param inc The number of elements the iterator is incremented.
// \param it The iterator to be incremented.
// \return The incremented iterator.
*/
template< typename Type  // Element type of the iterator
        , bool AF >      // Alignment flag of the iterator
inline const DenseIterator<Type,AF> operator+( ptrdiff_t inc, const DenseIterator<Type,AF>& it ) noexcept
{
   return DenseIterator<Type,AF>( it.base() + inc );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Subtraction between a DenseIterator and an integral value.
//
// \param it The iterator to be decremented.
// \param dec The number of elements the iterator is decremented.
// \return The decremented iterator.
*/
template< typename Type  // Element type of the iterator
        , bool AF >      // Alignment flag of the iterator
inline const DenseIterator<Type,AF> operator-( const DenseIterator<Type,AF>& it, ptrdiff_t dec ) noexcept
{
   return DenseIterator<Type,AF>( it.base() - dec );
}
//*************************************************************************************************


//*************************************************************************************************
/*!\brief Calculating the number of elements between two DenseIterator objects.
//
// \param lhs The left-hand side iterator.
// \param rhs The right-hand side iterator.
// \return The number of elements between the two iterators.
*/
template< typename Type  // Element type of the iterator
        , bool AF >      // Alignment flag of the iterator
inline ptrdiff_t operator-( const DenseIterator<Type,AF>& lhs, const DenseIterator<Type,AF>& rhs ) noexcept
{
   return lhs.base() - rhs.base();
}
//*************************************************************************************************

} // namespace blaze

#endif
